"""
PyTorch Lightning trainer for CLIP model.
"""

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional

from ..models.vision.clip import CLIP, CLIPConfig


class CLIPTrainer(pl.LightningModule):
    """
    Lightning module for training CLIP models.
    
    Handles contrastive learning between image and text modalities.
    """
    
    def __init__(
        self,
        config: CLIPConfig,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.2,
        warmup_steps: int = 10000,
        max_steps: int = 100000,
        betas: tuple = (0.9, 0.98),
        eps: float = 1e-6,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        
        self.model = CLIP(config)
        self.config = config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.betas = betas
        self.eps = eps
        
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through CLIP."""
        return self.model(images, input_ids, attention_mask, return_loss=True)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        images = batch['image']
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        
        # Forward pass
        outputs = self(images, input_ids, attention_mask)
        loss = outputs['loss']
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/logit_scale', outputs['logit_scale'], prog_bar=True, sync_dist=True)
        
        # Log learning rate
        lr = self.optimizers().param_groups[0]['lr']
        self.log('train/lr', lr, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        images = batch['image']
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask', None)
        
        # Forward pass
        outputs = self(images, input_ids, attention_mask)
        loss = outputs['loss']
        
        # Compute accuracy (top-1)
        logits = outputs['logits_per_image']
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean()
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/accuracy', acc, prog_bar=True, sync_dist=True)
        self.log('val/logit_scale', outputs['logit_scale'], sync_dist=True)
        
        return {'loss': loss, 'accuracy': acc}
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters: don't apply weight decay to bias and LayerNorm
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'bias' in name or 'norm' in name or 'ln' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ], lr=self.learning_rate, betas=self.betas, eps=self.eps)
        
        # Cosine learning rate schedule with warmup
        def lr_lambda(step):
            if step < self.warmup_steps:
                return step / self.warmup_steps
            else:
                progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }
    
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Log epoch metrics
        if hasattr(self.trainer, 'callback_metrics'):
            self.log('epoch', self.current_epoch, prog_bar=True)
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images for inference."""
        self.eval()
        with torch.no_grad():
            return self.model.encode_image(images, normalize=True)
    
    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode text for inference."""
        self.eval()
        with torch.no_grad():
            return self.model.encode_text(input_ids, attention_mask, normalize=True)
    
    def get_similarity(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute image-text similarity."""
        self.eval()
        with torch.no_grad():
            return self.model.get_similarity(images, input_ids, attention_mask)
